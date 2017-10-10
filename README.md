# ProbQA / Artificial Super-Neuron
I wasn't impressed by current artificial neural networks because even a single neuron out of 14 billions in a human brain has to be much more intelligent than a few bytes of information and a threshold function. Some life forms have no more than that - a single neuron - still they manage to do so much. So perhaps a neuron has processing power comparable to that of a single modern PC. That processing power/memory is used for storing large internal statistics and plans on how to interact with the other neurons.

So let me present an attempt to implement a single neuron using all the resources of a single PC: https://github.com/srogatch/ProbQA

The project has both Applied AI and General AI goals.

In terms of Applied AI goals, it's an expert system. Specifically, it's a probabilistic question-answering system: the program asks, the users answer. The minimal goal of the program is to identify what the user needs (a target), even if the user is not aware of the existence of such thing/product/service. It is just a backend in C++. It's up to the others to implement front-ends for their needs. The backend can be applied to something like this http://en.akinator.com/ , or for selling products&services in some internet-shops (as a chat-bot helping users to determine what they need, even if they can't formulate the keywords or even their desires specifically).

The AGI goal is to achieve the processing power of a single neuron of a human brain on a single PC. Interactions with billions of other computers should achieve human-level intelligence (AGI). You can join the effort in designing how one neuron would answer the questions of another neuron, and how their collective intelligence rises from cooperation.

To avoid confusion with the current Artificial Neural Network terminology, we may have to call these units "artificial super-neurons".
