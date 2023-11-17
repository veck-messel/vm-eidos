import src as ftdt

if __name__ == '__main__':
    grid = ftdt.Grid(shape=(20e-6, 16e-6, 1))
    
    grid[10e-6:14e-6, 2e-6:6e-6, 0] = ftdt.Object(permittivity=1.5**2, name='FirstObject')
    grid[2e-6:6e-6, 10e-6:14e-6, 0] = ftdt.Object(permittivity=1, name='SecondObject')

    grid[3e-6, 3e-6, 0] = ftdt.PointSource(name='FirstPointSource')
    grid[13e-6, 10e-6, 0] = ftdt.PointSource(name='SecondPointSource')

    grid[15e-6, :, 0] = ftdt.LineDetector(name='FirstLineDetector')
    
    grid[0:10, :, :] = ftdt.PML(name='PML x-low')
    grid[-10:, :, :] = ftdt.PML(name='PML x-high')
    grid[:, 0:10, :] = ftdt.PML(name='PML y-low')
    grid[:, -10:, :] = ftdt.PML(name='PML y-high')
    
    grid.run(total_time=500)
    grid.visualize(z=0, show=True)