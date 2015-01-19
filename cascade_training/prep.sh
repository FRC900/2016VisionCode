#! /bin/bash

# Create list of negative images, randomize the list
/bin/find negative_images -name \*.png > negatives.dat
/bin/find negative_images -name \*.jpg >> negatives.dat
shuf negatives.dat > temp.dat
mv temp.dat negatives.dat

# Create list of positive images
/bin/find positive_images -name \*.png > positives.dat
/bin/find positive_images -name \*.jpg >> positives.dat
# For each positive image, create a number of randomly rotated versions of that image
perl createtrainsamples.pl positives.dat negatives.dat . 9000 | tee foo.txt

# Merge each set of randomized versions of the images into one big .vec file
rm positives.vec
/bin/find . -name \*.vec > vectors.dat
mergevec/src/mergevec.exe vectors.dat ordered_positives.vec

# Randomize the order of those images inside the .vec file
mergevec/src/vec2img ordered_positives.vec samples%04d.png -w 20 -h 20 | shuf > info.dat
sed 's/$/ 1 0 0 20 20/' info.dat > random_info.dat
rm info.dat
mergevec/src/createsamples -info random_info.dat -vec positives.vec -num `wc -l info.dat` -w 20 -h 20
rm ordered_positives.vec

rm samples????.png
