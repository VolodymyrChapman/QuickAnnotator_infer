import configparser
import os

# Available projects
projs = os.listdir('projects')

# Print out available project dirs for user to select
print('Available projects:')

for i,e in enumerate(os.listdir('projects')):
    print(i, e)

# user selects desired project index
print('Please select index of desired project:')
i = int(input())

basedir = os.path.join('projects', projs[i])

# initialize a new config file:
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

# read options from selected config
config_path = os.path.join(basedir, 'config.ini')
print(config_path)
config.read(config_path)

# # display:
print(f'Config sections = {config.sections()}')