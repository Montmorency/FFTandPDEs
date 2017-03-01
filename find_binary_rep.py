N = 19
n = -1
while float(N)/(2.0**n) > 1.0:
  n += 1

if N==0:
  print '0'
elif N==1:
  print '1'
else:
  bin_string = ''
  while n > 0:
    n -= 1
    if (N-2**n) >= 0:
      print n
      print 'nT'
      bin_string += '1'
      N -= 2**n
    else:
      print 'nF'
      bin_string += '0'
  print bin_string
