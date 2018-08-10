while read -r line
do
    ./runall.sh $line
done < game.list
