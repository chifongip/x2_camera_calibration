from setuptools import find_packages, setup

setup(
    name='x2_camera_calibration',
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/x2_camera_calibration'],
        ),
        (
            'share/x2_camera_calibration',
            ['package.xml', 'README.md', 'LICENSE'],
        ),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='oscar',
    maintainer_email='oscar.ip@lscm.hk',
    description='Passive X2/Odin camera mounting calibration',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'capture = x2_camera_calibration.capture:main',
            'solve = x2_camera_calibration.offline:solve_main',
            'validate = x2_camera_calibration.offline:validate_main',
        ]
    },
)
