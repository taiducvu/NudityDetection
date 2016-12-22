### Activate the framework Tensorflow ###
source ~/tensorflow/bin/activate

### Run the model ###

python execute_model.py --checkpoint_dir='/home/taivu/workspace/NudityDetection/Dataset' --output_dir='/home/taivu/workspace/NudityDetection/Output' --eval_batch_size=100 --data_dir='/home/taivu/workspace/AddPic'


