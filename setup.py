from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Finetuning large transformer to summarizer many unread messages from telegram chats. Supports russian language. Access via telegram bot.',
    author='Hacker1337',
    license='MIT',
)
