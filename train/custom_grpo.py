# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging
import sys
import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from utils.configs import GRPOConfig, GRPOScriptArguments
from train.rewards import get_reward_funcs
from utils.callbacks import get_callbacks
from utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from utils.dataset_util import load_compeltion_dataset
from utils.model_util import get_tokenizer
from trainer.ActionKLTrainer import ActionKLTrainer

logger = logging.getLogger(__name__)

orig_torch_load = torch.load

def torch_wrapper(*args, **kwargs):
    logging.warning("[comfyui-unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled.")
    kwargs['weights_only'] = False

    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']


# def build_compute_metrics(tokenizer):
#     def compute_metrics(eval_pred):
#         print(eval_pred)
#         input("Press Enter to continue...")  # Debugging line to pause execution
#         predictions, labels = eval_pred
#         if hasattr(predictions, "shape") and len(predictions.shape) > 1:
#             predictions = predictions.argmax(axis=-1)
#         pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#         label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         exact_matches = [p.strip() == l.strip() for p, l in zip(pred_str, label_str)]
#         accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
#         return {"exact_match_accuracy": accuracy}
#     return compute_metrics

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    #验证wandb控制台是否关闭
    
    # Debug: 打印 local_rank
    rank = getattr(training_args, "local_rank", None)
    if rank is None:
        rank = getattr(training_args, "process_index", None)
    print(">>> local_rank/process_index:", rank)
    if torch.cuda.is_available() and rank is not None and rank != -1:
        torch.cuda.set_device(rank)

    ###############
    # Setup logging
    ###############
    #输出到控制台

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    
    # Load the dataset
    #dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    #print("model_init_kwargs:", training_args.model_init_kwargs)
    #input("Press Enter to continue...")  # Debugging line to pause execution
    # Load tokenizer
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    dataset = load_compeltion_dataset(script_args.dataset_name, script_args.language, script_args.prompt_mode)

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column, competion_column: str = script_args.dataset_completion_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        prompt_text = example[prompt_column]
        prompt.append({"role": "user", "content": f"```\n{prompt_text}\n```"})
        #compeltion = [{"role": "assistant", "content": example[competion_column]}]
        return {"prompt": prompt}
    
    dataset = dataset.map(make_conversation)
    # 输出一条数据
    if len(dataset) > 0:
        logger.info(f"Example conversation: {dataset['train'][0]['prompt']}")
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = ActionKLTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )
    #trainer.compute_metrics=build_compute_metrics(tokenizer)  # 这里传递,无效参数
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["code_completion"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
