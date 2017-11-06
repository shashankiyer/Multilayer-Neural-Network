# Multilayer-Neural-Network

The neural network is trained using stochastic gradient descent.

The Multilayer Neural Network calculates a set of weights to determine if a string is a special kind of palindrome.The neural network will operate on strings of length 40 over the alphabet {A,B,C,D}. It classifies strings as NONSTICK, 12-STICKY, 34-STICKY, 56-STICKY, 78-STICKY, STICK_PALINDROME.

A 40 character string is a stick palindrome if it can be written as the concatenations of two strings vw and v sticks with wR. It is k-sticky, if it can be written as a concatenation of three string uvw such that len(u)=k and u sticks with wR. The algorithm classifies a string as k(k+1)-STICKY if it is either k-sticky or k+1-sticky.

sticky_snippet_generator.py accepts 5 arguments namely: num_strings mutation_rate from_ends test_data.txt and are passed as python sticky_snippet_generator.py num_strings mutation_rate from_ends test_data.txt

sticky_snippet_net.py accepts 4 arguments namely: mode model_file data_folder mode can be either 'train', 'test' or '5_fold'. The first model trains the neural net and saves it to disk while the second reads a stored net and analyses it. The third performs 5-fold training and testing.
