def is_host_livia(hostname):
    return hostname in ["faraday", "pascal", "juna", "luna", "newton", "feynman", "turing"]

def custom_json_dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__name__
