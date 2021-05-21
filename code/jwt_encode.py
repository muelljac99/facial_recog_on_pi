import jwt
encoded_jwt = jwt.encode({"netid": "jacobam3"}, "ece498icc", algorithm="HS256")
print(encoded_jwt)

#print(jwt.decode(encoded_jwt, "ece498icc", algorithms=["HS256"]))