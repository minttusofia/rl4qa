# RL4QA
UCL MEng project

### Structure
- `/allennlp`: first experiment — applying AllenNLP PyTorch implementation of BiDAF to WikiHop
- `/baselines`: template baselines for WikiHop QA
- `/data/wikihop` (untracked): contains WikiHop
- `/ir`: search engine which builds an index for each question (in WikiHop) consisting of the 
support 
documents for that question. The index is kept in memory.
- `/playground`: frontend to interact with data using the search engine
- `/rc`: reading comprehension modules and utilities
- `/qa`: shared utilities for question processing and noun phrase extraction

### To run

##### Playground
1. Install jack (https://github.com/uclmr/jack).
2. Place WikiHop (train.json, dev.json, test.json) to `/data/wikihop`.
3. Create an index by running `python -m ir.search_engine` from the top 
level directory (this will take a while). Optionally, to only use a subset of data for faster 
development, add 
`--subset_size` followed by the desired size (e.g. `100`, or 100 questions).
4. Run `python -m playground.datareader` from the top level directory to start interacting with 
the data. Use the subset size from step 3.

##### Templates
1. See steps 1 - 3 under _Playground_ for setup. Build the index with
`k_most_common_relations_only = 6` (or less), as only these types have templates for now.
2. Run `python -m baselines.templates` from the top level directory to evaluate the template 
baseline on the data.
