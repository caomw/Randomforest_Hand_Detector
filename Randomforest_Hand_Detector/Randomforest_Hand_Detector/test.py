def dfs(u):
  print u
  if u < 1000:
    dfs(u + 1)
  else:
    print 'x'
    
    
dfs(0)