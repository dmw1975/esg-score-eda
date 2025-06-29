import os

class Location:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        
    def get_path(self, subdirectory, file_name=None):
        """
        Get the full path for a file or directory
        
        Parameters:
        subdirectory (str): Subdirectory name
        file_name (str, optional): File name
        
        Returns:
        str: Full path
        """
        # Create full path
        if file_name:
            full_path = os.path.join(self.base_dir, subdirectory, file_name)
        else:
            full_path = os.path.join(self.base_dir, subdirectory)
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(full_path) if file_name else full_path
        os.makedirs(directory, exist_ok=True)
        
        return full_path