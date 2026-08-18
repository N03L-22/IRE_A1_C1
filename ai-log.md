---
type: note
kind: ai-log
title: AI usage log — Assignment-1-Lexical-Semantic-Retrieval
---

# AI usage log — Assignment-1-Lexical-Semantic-Retrieval

Auto-captured prompts for this work item — a required IRE deliverable, and prompts
are **graded artifacts**.

Written by `Subjects/IRE/.claude/hooks/log-prompt.sh` on every prompt submitted
while working in this folder.

**Before submitting:** curate this file. Mark the decisive prompts with ⭐, delete
the noise, and add a short note on what worked and what failed — a log that shows a
real verification loop is the point, not a transcript.

---

### 2026-08-13 19:56 · `a172f976`

> <task-notification>
> <task-id>bhfyun60k</task-id>
> <tool-use-id>toolu_01AyYYaRDyzJvzmKuuBPyDjx</tool-use-id>
> <output-file>/tmp/claude-1000/-home-noel-Desktop-Obsidian-MTech-CSE-SEM3-Subjects-IRE/a172f976-711d-44d2-8dcd-c9f42bbcf7f4/tasks/bhfyun60k.output</output-file>
> <status>killed</status>
> <summary>Background command "Download EB-NeRD bundles and embeddings" was stopped</summary>
> </task-notification>

### 2026-08-15 00:16 · `e71a9c7f`

> 
> Ekstra Bladet News Recommendation Dataset - General License Terms
> Thank you! 
> 
> Please use the following links to access the dataset. We recommend opening the ones you want to download in different tabs or to print the page.
> 
> —————— ——————
> 
> - ebnerd_demo (20MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip  
> (*5,000 users)
> 
> —————— ——————
> 
> - ebnerd_small (80MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip
> (*50,000 users)
> 
> —————— ——————
> 
> - ebnerd_large (3.0GB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip 
> 
> - Articles (140MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip
> (Only download the articles from Large)
> 
> —————— ——————
> 
> - ebnerd_testset (1.5GB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip 
> 
> - Example of full submission file (220MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/predictions_large_random.zip
> (It’s all random predictions but this file will successfully upload to the leaderboard)
> 
> —————— ——————
> Artifacts:
> - Ekstra-Bladet-word2vec (133MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip 
> 
> - Ekstra_Bladet_image_embeddings (372MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip 
> 
> - Ekstra-Bladet-contrastive_vector (341MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip
> 
> - google-bert-base-multilingual-cased (344MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip 
> 
> - FacebookAI-xlm-roberta-base (341MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip 
> This content is neither created nor endorsed by Google. - Contact form owner - Terms of Service - Privacy Policy
> Does this form look suspicious? Report
> 
> Google Forms
> can this be downloaded faster? [Image #2] or are they still gate keeped.

### 2026-08-15 00:20 · `e71a9c7f`

> would computing on my own provide insights and be better? let it download large also. check the pdf for the info on embeddings? is it just for trial or must train yourself?

### 2026-08-15 00:32 · `e71a9c7f`

> i have the small and demo in the folders for the ebnerd. check if the architecture document is updated. clean simple explanation from foundations, simple to more advanced. and the the problems for which i need to decide

### 2026-08-15 00:41 · `e71a9c7f`

> data directory is not git ed and no files so far added ri8? check and add git if not. after reading, will proceed to code.

### 2026-08-15 00:53 · `e71a9c7f`

> yes thats mine, will finish soon. continue with the doc edits. for now lets plan this. make the core and gb as parseable args.

### 2026-08-15 00:58 · `e71a9c7f`

> lets use small for our prelim works

### 2026-08-15 19:55 · `e71a9c7f`

> on a side note, is it okay, if i use one terminal of claude for downloading and another for a different stage?

### 2026-08-18 15:00 · `e71a9c7f`

> list the requirements of the assignment in points, simple to understand

### 2026-08-18 15:12 · `e71a9c7f`

> we have the raw files ri8? check the status of the files. like if intact or any error, without unzipping? lets create a doc log for all these tasks, considerations, updates, things done, not done, why and etc. check the Q1.1 status. then for clean and parse, whats required? for temportal split, 80,10,10. generate the plan doc like in '/home/noel/Desktop/Obsidian/MTech_CSE/SEM3/Subjects/LMA/Projects/Individual-Small-LM/plan', one overall, and one for each phase. and once done, we can proceed.

### 2026-08-18 15:13 · `e71a9c7f`

> for now consider all phases, but go in depth till Q4. also in docs, for each design choice, provide alternatives and explain why and why not, features of each choice.

### 2026-08-18 15:17 · `e71a9c7f`

> use the small data set for now, instead of large. prep the models and stuff for this and then we can use the larger dataset

### 2026-08-18 15:28 · `e71a9c7f`

> check the large data sets in both.

### 2026-08-18 15:40 · `e71a9c7f`

> add those findings to the log.

### 2026-08-18 15:44 · `e71a9c7f`

> start phase 1, extract and readers

### 2026-08-18 16:29 · `e71a9c7f`

> should i install polar system wide? is pandas system wide or in our env?

### 2026-08-18 16:32 · `e71a9c7f`

> can i install polar in system wide and then import in venv?

### 2026-08-18 16:33 · `e71a9c7f`

> yes go ahead with all three steps

### 2026-08-18 16:42 · `e71a9c7f`

> then install in system and in venv

### 2026-08-18 16:42 · `e71a9c7f`

> no need just venv

### 2026-08-18 16:58 · `e71a9c7f`

> <task-notification>
> <task-id>bkylk2m85</task-id>
> <tool-use-id>toolu_01EbWg259J82bvxX3uqnTFPk</tool-use-id>
> <output-file>/tmp/claude-1000/-home-noel-Desktop-Obsidian-MTech-CSE-SEM3-Subjects-IRE/e71a9c7f-e274-41b2-9bdd-39440e305c8e/tasks/bkylk2m85.output</output-file>
> <status>completed</status>
> <summary>Background command "Install polars into venv" completed (exit code 0)</summary>
> </task-notification>

### 2026-08-18 17:05 · `e71a9c7f`

> continue

### 2026-08-18 17:20 · `e71a9c7f`

> continue with steps 3-5. make sure these are also logged.
