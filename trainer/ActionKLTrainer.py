from trl import GRPOTrainer, DPOTrainer
import torch


class ActionCovKLTrainer(GRPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        mask_logp = per_token_logps < per_token_logps.mean() 
        centered_advantage = advantages - advantages.mean()   
        centered_advantage = centered_advantage.unsqueeze(1) 
        centered_advantage = centered_advantage.expand_as(mask_logp)  # 扩展到与 mask_logp 相同的形状
        mask_adv = centered_advantage > 0
        mask_no_kl = mask_logp & mask_adv 
        mask_kl = ~mask_no_kl

        mask_logp_postive = per_token_logps > per_token_logps.mean()
        mask_over =  mask_logp_postive & mask_adv
        # 皮尔逊相关系数 logp_centered 与advantages
        cal_per_token_logps = per_token_logps -  per_token_logps.mean() #/ (per_token_logps.std() + 1e-8  # 标准化 logp
        # 计算相关系数
        correlation = cal_per_token_logps #* advantages.unsqueeze(1)
        # 根据mask_over把false的位置correlation变成0，true的保持不变
        correlation = correlation.masked_fill(~mask_over, 0.0)
        
        flat = correlation.flatten()
        positive_mask = flat > 0
        positive_values = flat[positive_mask]

        # 确保 k 不超过正值的实际数量
        k = max(1, int(positive_values.numel() * 0.02))  # 2%，至少保留1个
        k = min(k, positive_values.numel())  # 确保 k 不超过正值数量
        
        if k > 0 and positive_values.numel() > 0:
            topk_values, topk_indices = torch.topk(positive_values, k)
            # 构造全局mask
            global_mask = torch.zeros_like(flat, dtype=torch.bool)
            # 只在正值的位置里选topk
            positive_indices = positive_mask.nonzero(as_tuple=True)[0]
            topk_global_indices = positive_indices[topk_indices]
            global_mask[topk_global_indices] = True
            # 恢复原shape
            global_mask = global_mask.view_as(correlation)
            correlation = correlation.masked_fill(~global_mask, 1.0)
        else:
            correlation.zero_()

        # 让correlation没有梯度，只作为常量值
        #correlation = correlation.detach()
        #coef_correlation = torch.clamp(correlation, -1.0, 1.0)  # 确保相关系数在[-1, 1]范围内
        
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) *(2-correlation)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1) *(2-correlation)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl * mask_kl.float()

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clip ratio
        is_clipped = ((coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
            (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        )
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).nanmean().item())
        return loss


class Past_ActionCovKLTrainer(GRPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        mask_logp = per_token_logps < per_token_logps.mean() 
        centered_advantage = advantages - advantages.mean()   
        centered_advantage = centered_advantage.unsqueeze(1) 
        centered_advantage = centered_advantage.expand_as(mask_logp)  # 扩展到与 mask_logp 相同的形状
        mask_adv = centered_advantage > 0
        mask_no_kl = mask_logp & mask_adv 
        mask_kl = ~mask_no_kl

        mask_logp_postive = per_token_logps > per_token_logps.mean()
        mask_over =  mask_logp_postive & mask_adv
        # 皮尔逊相关系数 logp_centered 与advantages
        cal_per_token_logps = (per_token_logps -  per_token_logps.mean()) / (per_token_logps.std() + 1e-8)  # 标准化 logp
        # 计算相关系数
        correlation = cal_per_token_logps * advantages.unsqueeze(1)
        # 根据mask_over把false的位置correlation变成0，true的保持不变
        correlation = correlation.masked_fill(~mask_over, 0.0)
        
        flat = correlation.flatten()
        positive_mask = flat > 0
        positive_values = flat[positive_mask]

        # 确保 k 不超过正值的实际数量
        k = max(1, int(positive_values.numel() * 0.02))  # 2%，至少保留1个
        k = min(k, positive_values.numel())  # 确保 k 不超过正值数量
        
        if k > 0 and positive_values.numel() > 0:
            topk_values, topk_indices = torch.topk(positive_values, k)
            # 构造全局mask
            global_mask = torch.zeros_like(flat, dtype=torch.bool)
            # 只在正值的位置里选topk
            positive_indices = positive_mask.nonzero(as_tuple=True)[0]
            topk_global_indices = positive_indices[topk_indices]
            global_mask[topk_global_indices] = True
            # 恢复原shape
            global_mask = global_mask.view_as(correlation)
            correlation = correlation.masked_fill(~global_mask, 0.0)
        else:
            correlation.zero_()

        # 让correlation没有梯度，只作为常量值
        correlation = correlation.detach()
        coef_correlation = torch.clamp(correlation, -1.0, 1.0)  # 确保相关系数在[-1, 1]范围内
        
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) *(1-coef_correlation)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1) *(1-coef_correlation)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl * mask_kl.float()

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clip ratio
        is_clipped = ((coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
            (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        )
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).nanmean().item())
        return loss


    
class ActionKLTrainer(GRPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        mask_logp = per_token_logps < per_token_logps.mean() 
        
        centered_advantage = advantages - advantages.mean()   
        centered_advantage = centered_advantage.unsqueeze(1) 
        centered_advantage = centered_advantage.expand_as(mask_logp)  # 扩展到与 mask_logp 相同的形状
        mask_adv = centered_advantage > 0
        mask_no_kl = mask_logp & mask_adv 
        mask_kl = ~mask_no_kl

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl * mask_kl.float()

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clip ratio
        is_clipped = ((coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
            (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        )
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).nanmean().item())
        return loss