from glob import glob

from setuptools import setup

package_name = 'odometria'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    package_data={package_name: ['data/*.json']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='temurysega',
    maintainer_email='temurysega@users.noreply.github.com',
    description='Резервная одометрия трамвая по модели привода, колёсам и карте пути',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'odometry_node = odometria.node:main',
        'latency_probe = odometria.latency_probe:main',
    ]},
)
