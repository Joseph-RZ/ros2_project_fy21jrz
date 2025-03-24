from setuptools import find_packages, setup

package_name = 'ros2_project_fy21jrz'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cscajb',
    maintainer_email='x.wang16@leeds.ac.uk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'first_step = ros2_project_fy21jrz.first_step:main',
            'second_step = ros2_project_fy21jrz.second_step:main',
            'third_step = ros2_project_fy21jrz.third_step:main',
            'fourth_step = ros2_project_fy21jrz.fourth_step:main',
            'my_project = ros2_project_fy21jrz.my_project:main',
        ],
    },
)
