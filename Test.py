import numpy as np

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.pbest = position  # Personal best position
        self.gbest = position  # Global best position

    def update_position(self):
        self.position += self.velocity

    def update_velocity(self, pbest, gbest):
        c1 = 2
        c2 = 2
        w = 0.7  # Inertia weight

        r1 = np.random.random()
        r2 = np.random.random()

        velocity_pbest = c1 * r1 * (pbest - self.position)
        velocity_gbest = c2 * r2 * (gbest - self.position)

        self.velocity = w * self.velocity + velocity_pbest + velocity_gbest

    def calculate_fitness(self, GBW, CL, tan_PM, tan_60, I5):
        fitness = ((GBW * CL) * (tan_PM / tan_60)) / I5
        return fitness

def PSO(GBW, CL, tan_PM, tan_60, I5, num_particles, num_iterations):
    particles = []

    # Initialize particle positions and velocities
    for _ in range(num_particles):
        position = np.random.uniform(-1, 1, 2)  # Two-dimensional position vector
        velocity = np.random.uniform(-1, 1, 2)  # Two-dimensional velocity vector
        particle = Particle(position, velocity)
        particles.append(particle)

    # Main PSO loop
    for _ in range(num_iterations):
        gbest = particles[0].gbest  # Initialize gbest as the first particle's pbest

        for particle in particles:
            # Calculate fitness
            fitness = particle.calculate_fitness(GBW, CL, tan_PM, tan_60, I5)

            # Update personal best position
            if fitness > particle.pbest:
                particle.pbest = particle.position

            # Update global best position
            if fitness > gbest:
                gbest = particle.position

        # Update particle positions and velocities
        for particle in particles:
            particle.update_position()
            particle.update_velocity(particle.pbest, gbest)

    # Return the particle with the best fitness value
    best_particle = max(particles, key=lambda p: p.calculate_fitness(GBW, CL, tan_PM, tan_60, I5))
    return best_particle.position

# Example usage
GBW = 100  # MHz
CL = 0.8
tan_PM = 2
tan_60 = 1.732
I5 = -120  # dBm

num_particles = 10
num_iterations = 100

best_position = PSO(GBW, CL, tan_PM, tan_60, I5, num_particles, num_iterations)
print(f"Best position: {best_position}")
