#2D-classification
#all we need are: run_it025_classes.mrcs, run_it025_data.star

#class reader
python read_class.py #run_it025_classes.mrcs

#model re-train
python seperate_class.py #run_it025_data.star
sh runextractclass.sh
#python sort_class_by_whitevariance.py #use only .png
sh runtrain_type6.sh #use sorted_class.txt

#recenter
python read_class.py #run_it025_classes.mrcs
python get_recenter_transitions_of_each_class.py #use only .png
python scan_run_it_data_get_recenter_star.py [none/writeclass] #run_it025_data.star
