""" This module is responsible for initialising the environment variables including setting up the agent in the background. """

from dotenv import load_dotenv

def initialise():
    """  Load the environment variables from the .env file. """
    
    load_dotenv()
