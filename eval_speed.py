def speed_test(model, iterations = 100):
  model.eval()
  input = torch.randn(1,3,1024,2048).to(device)
  with torch.no_grad():
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(20):
      model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start
    
    print('========Speed Testing========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
      model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start
    latency = elapsed_time / iterations * 1000
  FPS = 1000 / latency
  print(FPS)
