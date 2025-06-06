from utils.eval_utils import (
    postprocess_code_lines,
    extract_identifiers,
    cal_edit_sim,
    remove_comments
)
b = "'\tpublic GithubApi provideAuthApi(Retrofit retrofit) {'"
a ='\nGithubApi provideGithubApi(Retrofit retrofit) {'
val = cal_edit_sim(a, b)
v = round(val,2)
print(v)