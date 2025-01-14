from setuptools import find_packages, setup

package_name = 'ros2_yolov5'

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
    maintainer='wie',
    maintainer_email='727894299@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_read_node = ros2_yolov5.data_read_node:main',
            'inference_node = ros2_yolov5.inference_node:main'
        ],
    },
)
