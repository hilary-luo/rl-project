from setuptools import setup

package_name = 'cas726_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hluo',
    maintainer_email='103377417+hilary-luo@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ai_explorer = cas726_project.ai_explorer:main',
            'frontier_explorer = cas726_project.frontier_explorer:main'
            'evaluator = cas726_project.evaluator:main'
        ],
    },
)
