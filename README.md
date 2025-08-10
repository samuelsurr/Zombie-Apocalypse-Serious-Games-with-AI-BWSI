# Team Splice
Worked under Rob Seater from MIT Lincoln Labratory creating a serious game from a baseline model and created the main RL and CNN mechanics. My teammates who worked alongside me were Jana, Aahan, Evan, and Sampreetha.
# SGAI_2025

# Run "python main.py" to run the game without data collection (use this to play if you haven't played before or just want to try and goof around)
# Run "python main.py --log True" for data collection purposes

Dependencies:
- Pandas
  - `pip3 install pandas`
- Numpy
  - `pip3 install numpy`
- PIL
  - `pip3 install pillow`
- TKinter
  - `pip3 install tk`
- Pytorch 
  - `pip3 install pytorch`
- OpenAI Gym (for RL)
  - `pip3 install gym==0.26.2`
- Timm (for SOTA computer vision models)
  - `pip3 install timm`
- Notion Client (for data collection)
  - `pip3 install notion-client`

To run, type ```python3 main.py``` in the terminal.


Alternatively:
Set up new Conda environment with:
```
conda config --add channels conda-forge 
conda create -n BWSI_2025 python=3.9 pandas numpy pillow tk pytorch pyyaml torchvision gym timm
```
