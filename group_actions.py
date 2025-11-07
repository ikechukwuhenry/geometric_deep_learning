import torch


def rotate_point(point, angle):
    """Rotate a 2D point by a given angle."""
    rotation_matrix = torch.tensor([
        [torch.cos(angle), -torch.sin(angle)],
        [torch.sin(angle), torch.cos(angle)]
    ])
    return torch.matmul(rotation_matrix, point)

def rotation_matrix(angle):
    """Generate a 2D rotation matrix for a given angle."""
    return torch.tensor([
        [torch.cos(angle), -torch.sin(angle)],
        [torch.sin(angle), torch.cos(angle)]
    ])

def rotations(angle, angle2):
    """Rotate a 2D point by a given angle."""
    r1 = rotation_matrix(angle)
    r2 = rotation_matrix(angle2)
    return torch.matmul(r1, r2)

def linear_rotation(angle, angle2):
    """Rotate a 2D point by a given angle."""
    sum_of_angle = angle + angle2
    return rotation_matrix(sum_of_angle)

# Example usage

identity_matrix = torch.eye(2)

theta = torch.tensor([90])  # No rotation
theta2 = torch.tensor([45])  # 90 degrees rotation
print(rotations(theta, theta2))

print(linear_rotation(theta, theta2))

rotated_matrix = rotate_point(identity_matrix, theta)
print(rotated_matrix)


r3 = torch.tensor([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
x_axis = torch.tensor([1, 0, 0])
print(torch.matmul(r3, x_axis))

print(torch.det(r3.float()))  # Should be 1 for rotation matrix

