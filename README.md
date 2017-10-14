# ProbQA / Artificial Super-Neuron
I wasn't impressed by current artificial neural networks because even a single neuron out of 14 billions in a human brain has to be much more intelligent than a few bytes of information and a threshold function. Some life forms have no more than that - a single neuron - still they manage to do so much. So perhaps a neuron has processing power comparable to that of a single modern PC. That processing power/memory is used for storing large internal statistics and plans on how to interact with the other neurons.

So let me present an attempt to implement a single neuron using all the resources of a single PC: https://github.com/srogatch/ProbQA

The project has both Applied AI and General AI goals.

In terms of Applied AI goals, it's an expert system. Specifically, it's a probabilistic question-answering system: the program asks, the users answer. The minimal goal of the program is to identify what the user needs (a target), even if the user is not aware of the existence of such thing/product/service. It is just a backend in C++. It's up to the others to implement front-ends for their needs. The backend can be applied to something like this http://en.akinator.com/ , or for selling products&services in some internet-shops (as a chat-bot helping users to determine what they need, even if they can't formulate the keywords or even their desires specifically).

The AGI goal is to achieve the processing power of a single neuron of a human brain on a single PC. Interactions with billions of other computers should achieve human-level intelligence (AGI). You can join the effort in designing how one neuron would answer the questions of another neuron, and how their collective intelligence rises from cooperation.

To avoid confusion with the current Artificial Neural Network terminology, we may have to call these units "artificial super-neurons".

Below is the learning curve of the program for matrix size 5 000 000: it's 1000 questions times 5 answer options for each question, times 1000 targets. In this experiment we train the program for binary search: the range of targets Tj is 0 to 999, and each question Qi is "How does your guess compare to Qi?". The answer options are 0 - "The guess is much lower than Qi", 1 - "The guess is a bit lower than Qi", 2 - "The guess exactly equals Qi", 3 - "The guess is a bit higher than Qi" and 4 - "The guess is much higher than Qi".

X-axis contains the number of questions asked&answered (up to 5 015 913). Y-axis contains for each 256 quizzes in a row the percentage of times the program correctly listed the guessed target among top 10 most probable targets. Note that testing is always on novel data: we first choose a random number, then let the program guess it by asking questions and getting answers from us, then either after the program has guessed correctly or asked more than 100 questions (meaning a failure), we teach the program, revealing it our selected random number.
![A diagram of training progress: precision over the number of questions asked&answered](https://raw.githubusercontent.com/srogatch/ProbQA/master/ProbQA/Notes/Metrics/TrainingProgress/LinearPriority.jpg)

I'm currently not sure about the best priority function, so the above result is for linear priority function. I plan to experiment with quadratic and cubic priority functions too: file ProbQA\ProbQA\PqaCore\CEEvalQsSubtaskConsider.cpp , [near the end of it currently](https://github.com/srogatch/ProbQA/blob/bb99aa26d1f27caa43a36b309a50beff6f8264ee/ProbQA/PqaCore/CEEvalQsSubtaskConsider.cpp#L111) .

There is also a flaw currently in the key theory, which makes the program stubborn (I think it's close to "overfitting" term of Machine Learning). After the program mistakenly selects some target as the most probable, it start asking such questions which let it stick to its mistake, rather than questions which would let the program see that other targets are more probable. Although it is what happens in life, technically it is an error in the key algorithm/theory behind the program.
