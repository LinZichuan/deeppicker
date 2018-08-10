job_picker=$1
jobname=$job_picker
ctf_choosed_jobname=$job_picker
ext_1=$job_picker
ext_2=$job_picker
class_use_ext_jobname=$job_picker
star_symbol=$2

Micrographs=$3
Micrographsstar=$4
particle_size=$5
op=$6 #classify or estimation

iter=$7
classK=$8
psi_step=$9
offset_range=${10}
offset_step=${11}

est="estimation"
cla="classify"
extract_size=$(echo "$particle_size*0.75"|bc|awk '{printf("%d", $1)}')
extract_size=$(echo "$extract_size*2"|bc|awk '{printf("%d", $1)}')
particle_diameter=$(echo "$particle_size*0.66"|bc|awk '{printf("%d", $1)}')
particle_diameter=$(echo "$particle_diameter*2"|bc|awk '{printf("%d", $1)}')
echo "extract_size=$extract_size"
echo "particle_diameter=$particle_diameter"


if [ "$op" = "$cla" ]; then
    echo "extract"
    #`which relion_preprocess_mpi` --i CtfFind/$ext_1/micrographs_ctf.star --coord_dir ./Import/$ext_2/ --coord_suffix $star_symbol.star --part_star Extract/$jobname/particles.star --part_dir Extract/$jobname/ --extract --extract_size 200 --norm --bg_radius 75 --white_dust -1 --black_dust -1 --invert_contrast  
    #`which relion_preprocess_mpi` --i CtfFind/$ext_1/micrographs_ctf.star --coord_dir ./Import/$ext_2/ --coord_suffix $star_symbol.star --part_star Extract/$jobname/particles.star --part_dir Extract/$jobname/ --extract --extract_size 280 --norm --bg_radius 75 --white_dust -1 --black_dust -1 --invert_contrast  
    # extract_size is 280pix [250A / 1.32 * 1.5(extra)]

    `which relion_preprocess_mpi` --i CtfFind/$ext_1/micrographs_ctf.star --coord_dir ./Import/$ext_2/ --coord_suffix $star_symbol.star --part_star Extract/$jobname/particles.star --part_dir Extract/$jobname/ --extract --extract_size $extract_size --norm --bg_radius 75 --white_dust -1 --black_dust -1 --invert_contrast  

    echo "2d-classify"
    mkdir -p Class2D/$jobname
    #`which relion_refine_mpi` --o Class2D/$jobname/run --i ./Extract/$class_use_ext_jobname/particles.star --dont_combine_weights_via_disc --pool 3 --ctf  --iter 25 --tau2_fudge 2 --particle_diameter 200 --K 50 --flatten_solvent  --zero_mask  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 8 --gpu ""

    #`which relion_refine` --o Class2D/$jobname/run --i Extract/$jobname/particles.star --dont_combine_weights_via_disc --pool 3 --ctf  --iter 25 --tau2_fudge 2 --particle_diameter 200 --K 50 --flatten_solvent  --zero_mask  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 16 --gpu ""
    #`which relion_refine` --o Class2D/$jobname/run --i Extract/$jobname/particles.star --dont_combine_weights_via_disc --pool 3 --ctf  --iter 25 --tau2_fudge 2 --particle_diameter 250 --K 50 --flatten_solvent  --zero_mask  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 16 --gpu ""
    # particle_diameter is 250A [particle_size ~= 250A / 1.32 = 189pix ]

    #`which relion_refine` --o Class2D/$jobname/run --i Extract/$jobname/particles.star --dont_combine_weights_via_disc --pool 3 --ctf  --iter 25 --tau2_fudge 2 --particle_diameter $particle_diameter --K 50 --flatten_solvent  --zero_mask  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 16 --gpu ""
    echo "`which relion_refine` --o Class2D/$jobname/run --i Extract/$jobname/particles.star --dont_combine_weights_via_disc --pool 3 --ctf  --iter $iter --tau2_fudge 2 --particle_diameter $particle_diameter --K $classK --flatten_solvent  --zero_mask  --oversampling 1 --psi_step $psi_step --offset_range $offset_range --offset_step $offset_step --norm --scale  --j 16 --gpu """
    `which relion_refine` --o Class2D/$jobname/run --i Extract/$jobname/particles.star --dont_combine_weights_via_disc --pool 3 --ctf  --iter $iter --tau2_fudge 2 --particle_diameter $particle_diameter --K $classK --flatten_solvent  --zero_mask  --oversampling 1 --psi_step $psi_step --offset_range $offset_range --offset_step $offset_step --norm --scale  --j 16 --gpu ""

    #piezo:190 -> 250A
    #zw18 :180 -> 237A

    #A = pix * 1.32

fi
