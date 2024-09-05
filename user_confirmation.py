import threading


# Generic function to ask user confirmation with a timeout and optional keys argument
def ask(params, keys=None, timeout=60):
    """
    Asks user for confirmation on a set of parameters and optionally allows them to provide new values.

    :param params: A dictionary of parameters (name: value) to be confirmed
    :param timeout: Time to wait for user input before proceeding with default values (in seconds)
    :param keys: Optional list of keys to ask about. If None, ask about all keys in params.
    :return: Updated parameters (name: value) as a dictionary
    """
    if keys:
        # Filter the params dictionary to include only the keys we're interested in
        params_to_ask = {k: v for k, v in params.items() if k in keys}
    else:
        params_to_ask = params  # If no keys are provided, use all params
    
    user_input = []
    
    def ask():
        confirmation = input(f"Do you want to proceed with these values? {params_to_ask} (y/n): ").lower()
        user_input.append(confirmation)
    
    # Start thread to ask for confirmation
    thread = threading.Thread(target=ask)
    thread.start()
    thread.join(timeout)  # Join with the timeout
    
    if thread.is_alive():
        print(f"No response within {timeout} seconds, proceeding with existing values.")
        return params  # Return the original params
    
    # If user responds 'n', ask for new values for the specified keys
    if user_input[0] == 'n':
        for param_name in params_to_ask:
            try:
                new_value = float(input(f"Please enter new value for {param_name}: "))
                params[param_name] = new_value  # Update the original params dictionary
            except ValueError:
                print(f"Invalid input for {param_name}! Keeping the original value: {params[param_name]}")
    
    return params

