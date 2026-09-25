from setuptools import setup

package_name = 'odometria'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_data={package_name: ['route.json', 'route_reverse.json']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='temurysega',
    maintainer_email='temurysega@users.noreply.github.com',
    description='Online model based tram odometry',
    license='MIT',
    entry_points={'console_scripts': ['odometry_node = odometria.node:main']},
)
