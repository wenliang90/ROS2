from setuptools import find_packages, setup

package_name = 'ros2_yolov5_distance'

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
            'rgbd_publish_node = ros2_yolov5_distance.rgbd_publish_node:main',
            'rgbd_pub_node = ros2_yolov5_distance.rgbd_publish_node:main',
            'det_dis = ros2_yolov5_distance.detection_disrance_node:main'
        ],
    },
)
