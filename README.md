# RL4QA
UCL MEng project

### Structure
- `/baselines`: template baselines for WikiHop QA
- `/data` (untracked): location for WikiHop and GloVe embeddings
- `/experiments`: defines jobs to be submitted to the UCL CS cluster
- `/ir`: search engine which builds an index for each question (in WikiHop) consisting of the 
support 
documents for that question. The index is kept in memory.
- `/playground`: frontend to interact with data using the search engine
- `/qa`: shared utilities for question processing and noun phrase extraction
- `/rc`: reading comprehension modules and utilities
- `/rl` : reinforcement learning agents
- `/shared` : utilities shared between `/rl` and `/baselines`

### To run

#### Playground
1. Install jack (see https://github.com/uclmr/jack).
2. Place WikiHop v1.1 (`train.json`, `dev.json`) under `/data/wikihop/v1.1/`.
3. Create an index by running `python -m ir.search_engine` from the top 
level directory (this will take a while). Optionally, to only use a subset of data for faster 
development, add 
`--subset_size` followed by the desired size (e.g. `100` for 100 questions), or 
`--k_most_common_only` to only include the k (e.g. `5`) most common WikiHop relation types.
4. Run `python -m playground.datareader` from the top level directory to start interacting with 
the data. Use the subset flags from step 3, and `--reader=bidaf` if using BiDAF.

#### Templates
1. See steps 1 - 3 under _Playground_ for setup.
2. To take advantage of Redis caching for faster reading comprehension answers, install Redis and
 run `redis-server rc/redis/redis.conf` to start a server. To skip caching, use the `--nocache` 
 flag in step 3.
3. Run `python -m baselines.templates` from the top level directory to evaluate the template 
baseline on the data. The set of templates to use can be specified with the `--templates_from_file`
flag.

#### Reinforce agent
1. See Steps 1 - 3 under _Playground_ for setup. To evaluate on dev data, repeat Step 3 with 
`--dev` to build a second index of dev data, or use the `--noeval` flag in Step 4 to work with
train 
data only.
2. Download `glove.6B.50d.txt` from https://nlp.stanford.edu/projects/glove/ and place it in
`/data/GloVe`.
3. To use caching, start redis-server as in step 2 in _Templates_, or use the `--nocache` flag in
 the next step.
3. Run `python -m rl.main` from the top level directory. Add `--random_agent` to evaluate a
random baseline agent, and/or `--run_id=<id>` to store checkpoints and TensorBoard summaries. Use
 the same subset flags (`--subset_size=<size>`, `--k_most_common_only=<k>`) that were used to build 
 the index. See `python -m rl.main --help` for a comprehensive list of arguments.
