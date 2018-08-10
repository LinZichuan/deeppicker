job_picker=$1
jobname=$job_picker
ctf_choosed_jobname=$job_picker
ext_1=$job_picker
ext_2=$job_picker
class_use_ext_jobname=$job_picker
star_symbol=$2


echo "import mrc and star"
mkdir -p Import/$jobname/
relion_star_loopheader rlnMicrographName > Import/$jobname/micrographs.star
ls Micrographs/*.mrc >> Import/$jobname/micrographs.star
cp --parents Micrographs/*$star_symbol.star Import/$jobname/
echo \"Micrographs/*.mrc\" > Import/$jobname/coords_suffix_pick.star
cp pick-result/*$star_symbol.star Micrographs/

echo "ctf estimation"
`which relion_run_ctffind_mpi` --i ./Import/$ctf_choosed_jobname/micrographs.star --o CtfFind/$jobname/ --CS 2.7 --HT 300 --AmpCnst 0.1 --XMAG 10000 --DStep 1.32 --Box 512 --ResMin 30 --ResMax 5 --dFMin 5000 --dFMax 50000 --FStep 500 --dAst 100 --use_gctf --gctf_exe ../../../Softwares/Gctf_v0.50/bin/Gctf-v0.50_sm_30_cu7.5_x86_64 --angpix 1.32 --ignore_ctffind_params --gpu "" 

echo "extract"
`which relion_preprocess_mpi` --i CtfFind/$ext_1/micrographs_ctf.star --coord_dir ./Import/$ext_2/ --coord_suffix $star_symbol.star --part_star Extract/$jobname/particles.star --part_dir Extract/$jobname/ --extract --extract_size 200 --norm --bg_radius 75 --white_dust -1 --black_dust -1 --invert_contrast  

echo "2d-classify"
mkdir -p Class2D/$jobname
#`which relion_refine_mpi` --o Class2D/$jobname/run --i ./Extract/$class_use_ext_jobname/particles.star --dont_combine_weights_via_disc --pool 3 --ctf  --iter 25 --tau2_fudge 2 --particle_diameter 200 --K 50 --flatten_solvent  --zero_mask  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 8 --gpu ""

`which relion_refine` --o Class2D/$jobname/run --i Extract/$jobname/particles.star --dont_combine_weights_via_disc --pool 3 --ctf  --iter 25 --tau2_fudge 2 --particle_diameter 200 --K 50 --flatten_solvent  --zero_mask  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 16 --gpu ""
