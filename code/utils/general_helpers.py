import uuid
import hashlib
import os
import re
import json
import json_repair
import pickle

from utils.openai_helpers import *


def save_to_pickle(a, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b



def local_recover_json(json_str):
    decoded_object = {}

    if '{' not in json_str:
        return json_str

    json_str = extract_json(json_str)

    try:
        decoded_object = json.loads(json_str)
        return decoded_object
    except Exception:
        try:
            decoded_object = json.loads(json_str.replace("'", '"'))
            return decoded_object
        except Exception:
            try:
                decoded_object = json_repair.loads(json_str.replace("'", '"'))

                for k, d in decoded_object.items():
                    dd = d.replace("'", '"')
                    decoded_object[k] = json.loads(dd)
                
                return decoded_object
            except:
                print(f"all json recovery operations have failed for {json_str}")
        
    return json_str


def read_file(text_filename):
    try:
        text_filename = text_filename.replace("\\", "/")
        with open(text_filename, 'r', encoding='utf-8') as file:
            text = file.read()
        status = True
    except Exception as e:
        text = ""
        print(f"WARNING ONLY - reading text file: {e}")
        status = False

    print(f"Success status: {status}. Reading file from full path: {os.path.abspath(text_filename)}")

    return text


def write_to_file(text, text_filename, mode = 'w'):
    try:
        text_filename = text_filename.replace("\\", "/")
        with open(text_filename, mode, encoding='utf-8') as file:
            file.write(text)

        status = f"Writing file to full path: {os.path.abspath(text_filename)}"
        print(status)
        
    except Exception as e:
        status = f"SERIOUS ERROR: Error writing text to file: {e}"
        print(status)

    return status


def generate_random_uuid():
    return str(uuid.uuid4())


def generate_uuid_from_string(input_string):
    # Create a SHA-1 hash of the input string
    hash_object = hashlib.sha1(input_string.encode())
    # Use the first 16 bytes of the hash to create a UUID
    return str(uuid.UUID(bytes=hash_object.digest()[:16]))


def get_file_md5(file_name):
    with open(file_name, 'rb') as file_obj:
        file_contents = file_obj.read()
        md5 = hashlib.md5(file_contents).hexdigest()
        return str(md5)
    


def list_files_in_directory(directory_path):
    """
    Returns a list of all files in the given directory.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        list: A list of filenames in the directory.
    """
    try:
        # List all files in the directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

