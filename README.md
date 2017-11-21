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
1. Place WikiHop (train.json, dev.json, test.json) to `/data/wikihop`
2. Run `python -m playground.datareader` from the top level directory
