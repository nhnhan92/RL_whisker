export SOFA_ROOT=/home/truongdo/sofa/build/

export PYTHONPATH=/home/truongdo/sofa/build/lib/python3/site-packages:$PYTHONPATH

# python test_env.py -e trunk-v0 -ep 100 -s 100

python rl.py -e trunk-v0 -a PPO -ne 8