# Download the checkpoint and put it into models/research/object_detection/test_data/
rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8*
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
mv ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint experiments/pretrained_model/
rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8*