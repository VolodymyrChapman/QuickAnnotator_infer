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
i = input()

basedir = projs[i]

# initialize a new config file:
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

# read options from selected config
config.read(os.path.join('project', basedir))

# # display:
# print(f'Config sections = {config.sections()}')