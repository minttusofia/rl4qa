# RL4QA
UCL MEng project

### Structure
- `/allennlp`: first experiment — applying AllenNLP PyTorch implementation of BiDAF to WikiHop
- `/data/wikihop` (untracked): contains WikiHop
- `/ir`: search engine which builds an index for each question (in WikiHop) consisting of the 
support 
documents for that question. The index is kept in memory.
- `/playground`: frontend to interact with data using the search engine.

### To run
##### Playground
1. Install jack (https://github.com/uclmr/jack)
2. Place WikiHop (train.json, dev.json, test.json) to `/data/wikihop`
3. Create an index by running `python -m ir.search_engine` from the top 
level directory (this will take a while). Optionally, to only use a subset of data for faster 
development, add 
`--subset_size` followed by the desired size (e.g. `100`, or 100 questions).
4. Run `python -m playground.datareader` from the top level directory to start interacting with 
the data.
