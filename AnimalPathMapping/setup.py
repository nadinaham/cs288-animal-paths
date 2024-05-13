from setuptools import setup, find_packages
  
with open("README.md", "r") as fh:
    description = fh.read()
  
setup(
    name="AnimalPathMapping",
    version="0.0.1",
    author="Nadine Han, Samantha Marks, and Yiyi Wang",
    author_email="nhan@college.harvard.edu, samanthamarks@college.harvard.edu, yiyiwang@fas.harvard.edu",
    packages=find_packages(),
    description="Python package for automatically labeling animal paths173572 in Davies lab imagery of conservation sites in Africa.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/nadinaham/cs288-animal-paths",
    license='none',
    python_requires='==3.12.2',
    install_requires=[]
)