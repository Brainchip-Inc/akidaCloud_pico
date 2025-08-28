cd examples
mkdir -p models
cd models

wget https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_eye_buffer_i8_w8_a8.h5
wget https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_jester_buffer_i8_w8_a8.h5
wget https://data.brainchip.com/models/AkidaV2/centernet/centernet_akidanet18_voc_384_i8_w8_a8.h5

find . -maxdepth 1 -type f \( -name "*.h5" \) -exec cnn2snn convert -m  {} \;
rm *.h5


# Get datasets for examples
cd ..
cd datasets

wget https://data.brainchip.com/dataset-mirror/voc/test_20_classes.tfrecord

wget https://data.brainchip.com/dataset-mirror/jester/jester_subset.tar.gz
tar -xzf ./jester_subset.tar.gz  jester_subset/jester-v1-labels.csv 
rm ./jester_subset.tar.gz