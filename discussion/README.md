# Belle selective MC dataset

Add link here to all publications/talks.

## Repository contents

* Overall description of dataset goal
* Dataset files
* Description of dataset file contents
* Example scripts
* (Optional) Saved model

### Overall description of dataset goal

This should explain the abstract goal of the project, no Belle jargon included.
And if possible no physics jargon either (description of the variables not included).

### Dataset files

There are three components needed:
1. Model input
2. Labels
3. Decorrelation variables
These can be split further but in the end this is what users need.

The model input includes node features and node connections (edges).
We have no edge features or weights.
Labels are simply one binary label per graph.
We do not have node or edge labels.
Decorrelation variables are whole-graph features.
These cannot be used as model input, this needs to be made clear to users.

There are several ways to format the data, I've added my thoughts on each below.
They basically boil down to saving the adjacency matrix or edge lists.
The goal should be simplicity of use, and I think this means providing a familiar
format for people using existing graph libraries like [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [Graph Nets](https://github.com/deepmind/graph_nets) and [Deep Graph Library](https://github.com/dmlc/dgl).
I only know about PyTorch Geometric, that expects edge lists as input.

#### Pandas dataframes

This is what we've got at the moment.
It puts everything as a 2D array and uses a MultiIndex to separate events.

##### Pros

* Minimal work to produce
* Familiar enough format

##### Cons

* Requires people use Pandas to at least load them (slow)
* Requires some data manipulation to produce edge lists.

#### Numpy arrays

These could either be adjacency matrices or edge lists.
I'm not sure how to deal with the different sized graphs without padding.

##### Pros

* Also minimal work to produce from our Pandas dataframes
* This is pretty much a ready-to-use format.

##### Cons

* Need to deal with the different sized graphs in a way that won't mess up trainings.

#### Text files ([TU Dortmund Graph Kernel style](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets))

This is currently a standard way to provide datasets, though obviously not as size-efficient.
It provides the edges, edge labels, node labels, node attributes, graph labels, graph indicators (which lines in each file belong to which graph), and a readme all as separate text files.
In theory there's no reason we can provide these as 2D numpy arrays which would save space.

##### Pros

* Familiar
* Solves the issue of storing different sized graphs
* I already have a script to read these and convert to padded adjacency matrices

##### Cons

* Size inefficient (see above)
* Gonna be a bitch to code


### Description of dataset file contents

Will depend on the format we choose.

### Example scripts

Need to provide example of the following:
* Loading the data
* The metric used to score each dataset
* (Optional) Visualising the data

### (Optional) Saved model

This is tricky, if our model has custom layer.
In reality it should be a model made using one of the standard graph libraries.
Maybe we leave this as a future addition for now.
