# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Example commands to run the train and predict.py files:
For train.py:
python train.py flowers
python train.py flowers --save_dir checkpoint.pth --arch "vgg16" --learning_rate 0.0001 --hidden 600 --epoch 10 --gpu gpu

For predict.py:
python predict.py flowers/test/19/image_06196.jpg checkpoint.pth
python predict.py flowers/test/19/image_06196.jpg checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu gpu
