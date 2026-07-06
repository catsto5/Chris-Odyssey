def generate_traj(m, T, sigma, c):
  np.random.seed(1)
  eta = np.random.normal(loc=0, scale=sigma, size=(m, T)) #generate i.i.d. N(0,\sigma^2)
  x = np.zeros((m, T+1))  #initial state x_0=0
  for i in range(m):
    for t in range(T):
      x[i,t+1]=f(x[i,t],c_l)+eta[i,t]
  return x
