from setuptools import find_packages, setup


package_name = "aigp_bag_tools"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="AI-GP Team",
    maintainer_email="dev@example.com",
    description="Convert AI-GP FlightSim JSONL logs into ROS 2 Humble rosbags.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "jsonl_to_rosbag = aigp_bag_tools.jsonl_to_rosbag:main",
            "verify_bag = aigp_bag_tools.verify_bag:main",
            "rosbag_to_dataset = aigp_bag_tools.rosbag_to_dataset:main",
        ],
    },
)
