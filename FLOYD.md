
login:
floyd login -u jordiburgos

create data:
floyd data init jordiburgos/algo

upload:
floyd data upload

train:
floyd run --data jordiburgos/datasets/numerai108/3:data python train.py

train GPU:
floyd run --gpu --data jordiburgos/datasets/numerai108/3:data python train.py

predict:
floyd run --data jordiburgos/datasets/numerai108/1:data --data jordiburgos/projects/numerai/17/output:/model python predict.py

predict GPU:
floyd run --gpu --data jordiburgos/datasets/numerai108/3:data --data jordiburgos/projects/numerai/20/output:/model python predict.py

explore:
floyd run --data jordiburgos/datasets/numerai108/3:data python explore.py
