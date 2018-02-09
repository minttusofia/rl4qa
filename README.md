# RL4QA
UCL MEng project

### Structure
- `/allennlp`: first experiment — applying AllenNLP PyTorch implementation of BiDAF to WikiHop
- `/baselines`: template baselines for WikiHop QA
- `/data` (untracked): location for WikiHop and GloVe embeddings
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
2. Place WikiHop v1.1 (train.json, dev.json) under `/data/wikihop/v1.1/`.
3. Create an index by running `python -m ir.search_engine` from the top 
level directory (this will take a while). Optionally, to only use a subset of data for faster 
development, add 
`--subset_size` followed by the desired size (e.g. `100` for 100 questions), or 
`--k_most_common_only` to only include the k (e.g. `5`) most common WikiHop relation types.
4. Run `python -m playground.datareader` from the top level directory to start interacting with 
the data. Use the subset flags from step 3.

#### Templates
1. See steps 1 - 3 under _Playground_ for setup. Build the index with
`k_most_common_only=6` (or less), as only these types have templates for now.
2. To take advantage of Redis caching for faster reading comprehension answers, install Redis and
 run `redis-server rc/redis/redis.conf` to start a server. To skip caching, use the `--nocache` 
 flag in step 3.
3. Run `python -m baselines.templates` from the top level directory to evaluate the template 
baseline on the data.

#### Reinforce agent
1. See Steps 1 - 3 under _Playground_ for setup.
2. To use caching, start redis-server as in step 2 in _Templates_, or use the `--nocache` flag in
 the next step.
3. Run `python -m rl.agent` from the top level directory. Add `--random_agent` to evaluate a 
random baseline agent, and/or `--run_id=<id>` to store checkpoints and TensorBoard summaries. Use
 the same subset flags (`--subset_size=<size>`, `--k_most_common_only=<k>`) that were used to build 
 the index.
