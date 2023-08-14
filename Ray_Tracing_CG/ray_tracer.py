import argparse
from PIL import Image
import numpy as np
import math

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

import time


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # Create an array of zeros for the image
    image_array = np.zeros((args.width, args.height, 3))

    # Calculate the camera parameters and image center
    camera.look_at = np.array(camera.look_at) - np.array(camera.position)
    camera.look_at = camera.look_at / np.linalg.norm(camera.look_at)
    camera.up_vector = camera.up_vector / np.linalg.norm(camera.up_vector)
    image_center = camera.position + np.array(camera.look_at) * camera.screen_distance

    # Calculate the right and up vectors
    v_right = np.cross(camera.look_at, camera.up_vector)
    v_right = v_right / np.linalg.norm(v_right)
    v_up = np.cross(v_right, camera.look_at)
    v_up = v_up / np.linalg.norm(v_up)

    # Calculate the ratio of the screen width to the image width
    ratio = camera.screen_width / args.width

    # Loop through each pixel in the image
    for i in range(args.height):
        for j in range(args.width):
            # Calculate the ray for the current pixel
            ray = image_center - v_right * ratio * (j - math.floor(args.width / 2)) - v_up * ratio * (
                    i - math.floor(args.height / 2)) - camera.position
            ray = ray / np.linalg.norm(ray)

            # Calculate the color for the current pixel
            ray_tracer(ray, i, j, image_array, objects, scene_settings, camera.position, 1)

    # Convert the image array to integers in the range [0, 255]
    for i in range(args.height):
        for j in range(args.width):
            for k in range(3):
                image_array[i][j][k] = int(image_array[i][j][k])
    image_array = image_array.clip(0, 255)

    # Save the output image
    save_image(image_array)


def ray_tracer(ray, i, j, image_array, objects, scene_settings, origin_point, depth):
    # check if the recursion depth has been exceeded
    if depth > scene_settings.max_recursions:
        return np.array([0, 0, 0])
    # Initialize the closest surface and closest intersection distance
    closest_surface = (None, float('inf'))
    closest_intersection_distance = float('inf')
    # Loop through each surface in the scene
    for surface in objects:
        # Skip objects which are not actually surfaces
        if type(surface) in [Light, Material]:
            pass
        elif type(surface) == Sphere:
            # Calculate the coefficients of the quadratic equation
            coefficients = [1, np.dot(2 * ray, np.array(origin_point) - np.array(surface.position)),
                            np.linalg.norm(np.array(origin_point) - np.array(
                                surface.position)) ** 2 - surface.radius ** 2]
            # Define the discriminant and check if its non-negative to avoid imaginary roots
            discriminant = (coefficients[1] ** 2) - (4 * coefficients[0] * coefficients[2])
            if discriminant >= 0:
                # Calculate the roots of the quadratic equation, the results are distances from the origin point
                roots = [(-coefficients[1] - math.sqrt(discriminant)) / (2 * coefficients[0]),
                         (-coefficients[1] + math.sqrt(discriminant)) / (2 * coefficients[0])]
                # Check if the roots are positive and less than the closest intersection distance
                for t in roots:
                    if 0.00001 < t < closest_intersection_distance:
                        # Calculate the point of intersection, set the closest intersection distance, and set the closest surface
                        point_of_intersection = origin_point + t * ray
                        closest_intersection_distance = t
                        closest_surface = (surface, point_of_intersection)

        elif type(surface) == InfinitePlane:
            # Normalize the surface normal
            surface_normal = np.array(surface.normal)
            surface_normal = surface_normal / np.linalg.norm(surface_normal)
            # Check if the ray is parallel to the plane, if not, calculate the distance to the plane
            if np.dot(ray, surface_normal) != 0:
                # Calculate the distance to the plane
                t = -(np.dot(origin_point, surface_normal) - surface.offset) / np.dot(ray, surface_normal)
                # Check if the distance is positive and less than the closest intersection distance
                if 0.00001 < t < closest_intersection_distance:
                    # Calculate the point of intersection, set the closest intersection distance, and set the closest surface
                    point_of_intersection = origin_point + t * ray
                    closest_intersection_distance = t
                    closest_surface = (surface, point_of_intersection)

        elif type(surface) == Cube:

            # Calculate the center and edge length of the cube
            center = surface.position
            edge_length = surface.scale

            # Create a vector that represents the size of the sides of the cube
            cube_size_vector = np.array([edge_length / 2, edge_length / 2, edge_length / 2])

            # Calculate the exit points for the x-axis
            min_exit = center - cube_size_vector
            max_exit = center + cube_size_vector

            x_min, x_max = sorted([(min_exit[0] - origin_point[0]) / ray[0], (max_exit[0] - origin_point[0]) / ray[0]])
            y_min, y_max = sorted([(min_exit[1] - origin_point[1]) / ray[1], (max_exit[1] - origin_point[1]) / ray[1]])
            z_min, z_max = sorted([(min_exit[2] - origin_point[2]) / ray[2], (max_exit[2] - origin_point[2]) / ray[2]])

            t_min = max(x_min, y_min, z_min)
            t_max = min(x_max, y_max, z_max)

            # Check if the ray intersects the cube
            if t_min < t_max:
                # Check if the intersection point is closer than the previous closest intersection point
                if 0.00001 < t_min < closest_intersection_distance:
                    # Calculate the point of intersection, set the closest intersection distance, and set the closest surface
                    closest_intersection_distance = t_min
                    closest_surface = (surface, origin_point + closest_intersection_distance * ray)

    # Check if the closest surface is None, if so return the background color (because there is no intersection)
    if closest_surface[0] is None:
        # Check if the depth is 1, if so set the pixel to the background color and return the background color
        if depth == 1:
            image_array[i][j] = np.array(scene_settings.background_color) * 255
        return np.array(scene_settings.background_color) * 255
    # If the closest surface is not None (there is an intersection), calculate the color of the pixel
    else:
        # Calculate the normal of the closest surface
        if type(closest_surface[0]) == Sphere:
            normal = closest_surface[1] - closest_surface[0].position
            normal = normal / np.linalg.norm(normal)

        # Calculate the normal of the closest surface
        elif type(closest_surface[0]) == InfinitePlane:
            normal = closest_surface[0].normal
            normal = normal / np.linalg.norm(normal)

        # Calculate the normal of the closest surface (create 6 planes and check which one is the closest) and set the normal
        # to the normal of the closest plane
        elif type(closest_surface[0]) == Cube:
            center = closest_surface[0].position
            edge_length = closest_surface[0].scale
            closet_plane = -1
            minimal_distance = float('inf')
            if abs(closest_surface[1][0] - (center[0] + edge_length / 2)) < minimal_distance:
                closet_plane = 0
                minimal_distance = abs(closest_surface[1][0] - (center[0] + edge_length / 2))

            if abs(closest_surface[1][0] - (center[0] - edge_length / 2)) < minimal_distance:
                closet_plane = 1
                minimal_distance = abs(closest_surface[1][0] - (center[0] - edge_length / 2))

            if abs(closest_surface[1][1] - (center[1] + edge_length / 2)) < minimal_distance:
                closet_plane = 2
                minimal_distance = abs(closest_surface[1][1] - (center[1] + edge_length / 2))

            if abs(closest_surface[1][1] - (center[1] - edge_length / 2)) < minimal_distance:
                closet_plane = 3
                minimal_distance = abs(closest_surface[1][1] - (center[1] - edge_length / 2))

            if abs(closest_surface[1][2] - (center[2] + edge_length / 2)) < minimal_distance:
                closet_plane = 4
                minimal_distance = abs(closest_surface[1][2] - (center[2] + edge_length / 2))

            if abs(closest_surface[1][2] - (center[2] - edge_length / 2)) < minimal_distance:
                closet_plane = 5
                minimal_distance = abs(closest_surface[1][2] - (center[2] - edge_length / 2))

            if closet_plane == 0:
                normal = np.array([1, 0, 0])
            elif closet_plane == 1:
                normal = np.array([-1, 0, 0])
            elif closet_plane == 2:
                normal = np.array([0, 1, 0])
            elif closet_plane == 3:
                normal = np.array([0, -1, 0])
            elif closet_plane == 4:
                normal = np.array([0, 0, 1])
            elif closet_plane == 5:
                normal = np.array([0, 0, -1])
            normal = normal / np.linalg.norm(normal)

        # Calculate the view vector
        view = -(origin_point - closest_surface[1])
        view = view / np.linalg.norm(view)

        # Get the material of the closest surface
        material_index = closest_surface[0].material_index
        material_counter = 0
        surface_material = None
        for object in objects:
            if type(object) == Material:
                material_counter += 1
                if material_counter == material_index:
                    surface_material = object
                    break

        # Get the material properties
        material_diffuse = surface_material.diffuse_color
        material_specular = surface_material.specular_color
        # Initialize the return color
        return_color = np.zeros(3)

        # Go for every light in the scene
        for light in objects:
            # Check if the object is a light
            if type(light) is not Light:
                continue
            # Check if the light is a point light
            else:

                shadow_intensity = light.shadow_intensity

                # Calculate the vector from the intersection point to the light and normalize it
                intersection_to_light = light.position - closest_surface[1]
                intersection_to_light = intersection_to_light / np.linalg.norm(intersection_to_light)

                # Calculate the vector from the intersection point to the reflected light and normalize it
                intersection_to_reflected_light = 2 * np.dot(intersection_to_light,
                                                             normal) * normal - intersection_to_light
                intersection_to_reflected_light = intersection_to_reflected_light / np.linalg.norm(
                    intersection_to_reflected_light)

                # Calculate the reflected ray
                reflected_ray = ray - 2 * np.dot(ray, normal) * normal
                reflected_ray = reflected_ray / np.linalg.norm(reflected_ray)

                # Get the grid width
                grid_width = light.radius

                # Get the grid ratio by dividing the grid width by the number of shadow rays (getting the size of each
                # grid cell)
                grid_ratio = grid_width / scene_settings.root_number_shadow_rays

                # Create a vector that is different from the intersection to light vector
                rand_vector = np.array(
                    [intersection_to_light[0], intersection_to_light[1], intersection_to_light[2] + 1])
                # Normalize the vector
                rand_vector = rand_vector / np.linalg.norm(rand_vector)
                # Get a vector that is perpendicular to the intersection to light vector and the random vector
                light_v_up = np.cross(-intersection_to_light, rand_vector)
                # Normalize the vector
                light_v_up = light_v_up / np.linalg.norm(light_v_up)
                # Get a vector that is perpendicular to the intersection to light vector and the light v up vector
                light_v_right = np.cross(-intersection_to_light, light_v_up)
                # Normalize the vector
                light_v_right = light_v_right / np.linalg.norm(light_v_right)

                # Initialize the shadow rays count
                shadow_rays_count = 0

                # Go for every grid cell
                for x in range(int(scene_settings.root_number_shadow_rays)):
                    for y in range(int(scene_settings.root_number_shadow_rays)):
                        # Calculate the point on the grid (with a random offset)
                        point_on_grid = light.position - light_v_right * grid_ratio * (
                                x - math.floor(
                            scene_settings.root_number_shadow_rays / 2)) - light_v_up * grid_ratio * (y - math.floor(
                            scene_settings.root_number_shadow_rays) / 2) + ((
                                                                                    np.random.rand() - 0.5) * grid_ratio * light_v_right +
                                                                            (
                                                                                    np.random.rand() - 0.5) * grid_ratio * light_v_up)
                        # Calculate the ray from the grid cell to the intersection point and normalize it
                        grid_ray = - (point_on_grid - closest_surface[1])
                        grid_ray = grid_ray / np.linalg.norm(grid_ray)

                        # Check if the grid ray hits the intersection point
                        is_hit = ray_tracer_shadow(grid_ray, objects, closest_surface[1], point_on_grid)
                        # If the grid ray hits the intersection point, increase the shadow rays count
                        if is_hit:
                            shadow_rays_count += 1

                # Calculate the light intensity
                light_intensity = (1 - shadow_intensity) * 1 + shadow_intensity * (
                        shadow_rays_count / (scene_settings.root_number_shadow_rays ** 2))

                # Calculate the diffusion and specular for the current light
                diffusion_and_specular = (np.array(material_diffuse) * np.dot(normal, intersection_to_light) + \
                                          np.array(material_specular) * np.dot(view,
                                                                               intersection_to_reflected_light) ** surface_material.shininess) * light_intensity * light.specular_intensity

                # Add the diffusion and specular of the current light to the return color
                return_color += np.array(diffusion_and_specular) * \
                                (1 - surface_material.transparency) * np.array(light.color) * 255

        # Calculate the recursion color
        recursion_color = ray_tracer(reflected_ray, i, j, image_array, objects, scene_settings, closest_surface[1],
                                     depth + 1)

        # If the surface is not transparent, add the recursion color to the return color
        if surface_material.transparency == 0:
            return_color += np.array(surface_material.reflection_color) * recursion_color

        # If the surface is transparent, calculate the transparency color and add it to the return color (with the
        # recursion color)
        else:
            transparency_color = ray_tracer(ray, i, j, image_array, objects, scene_settings, closest_surface[1],
                                            depth + 1)
            return_color += transparency_color * np.array(
                surface_material.transparency) + np.array(surface_material.reflection_color) * recursion_color

        # If the depth is 1, set the color of the pixel to the return color (this is the first ray)
        if depth == 1:
            image_array[i, j] = return_color
        return return_color


# Function that checks if a shadow ray hits a specific point, very similar in code to the ray_tracer function first part
def ray_tracer_shadow(ray, objects, original_intersection_point, point_on_grid):
    # Initialize the closest surface
    closest_surface = (None, float('inf'))
    closest_intersection_distance = float('inf')
    point_of_intersection = None
    # Go for every object in the scene
    for surface in objects:
        # Check if the object is a light or a material, if it is, do nothing
        if type(surface) in [Light, Material]:
            pass
        elif type(surface) == Sphere:
            # Calculate the coefficients of the quadratic equation
            coefficients = [1, np.dot(2 * ray, np.array(point_on_grid) - np.array(surface.position)),
                            np.linalg.norm(np.array(point_on_grid) - np.array(
                                surface.position)) ** 2 - surface.radius ** 2]
            # Calculate the discriminant of the quadratic equation and check if its not negative
            discriminant = (coefficients[1] ** 2) - (4 * coefficients[0] * coefficients[2])
            if discriminant >= 0:
                # Calculate the roots of the quadratic equation, the result is a list of 2 roots which are the
                # distances from the point on the grid to the intersection points
                roots = [(-coefficients[1] - math.sqrt(discriminant)) / (2 * coefficients[0]),
                         (-coefficients[1] + math.sqrt(discriminant)) / (2 * coefficients[0])]
                # Go for every root
                for t in roots:
                    # Check if the root is positive and smaller than the closest intersection distance
                    if 0.00001 < t < closest_intersection_distance:
                        # Calculate the point of intersection and set the closest intersection distance to the root
                        point_of_intersection = point_on_grid + t * ray
                        closest_intersection_distance = t
                        closest_surface = (surface, point_of_intersection)

        elif type(surface) == InfinitePlane:
            # Calculate the surface normal and normalize it
            surface_normal = np.array(surface.normal)
            surface_normal = surface_normal / np.linalg.norm(surface_normal)
            # Check if the ray is not parallel to the surface
            if np.dot(ray, surface_normal) != 0:
                # Calculate the distance from the point on the grid to the intersection point
                t = -(np.dot(point_on_grid, surface_normal) - surface.offset) / np.dot(ray, surface_normal)
                # Check if the distance is positive and smaller than the closest intersection distance
                if 0.00001 < t < closest_intersection_distance:
                    # Calculate the point of intersection and set the closest intersection distance to the distance
                    point_of_intersection = point_on_grid + t * ray
                    closest_intersection_distance = t
                    closest_surface = (surface, point_of_intersection)

        elif type(surface) == Cube:

            center = surface.position
            edge_length = surface.scale

            # Create a vector that represents the size of the sides of the cube
            cube_size_vector = np.array([edge_length / 2, edge_length / 2, edge_length / 2])

            # Calculate the exit points for the x-axis
            min_exit = center - cube_size_vector
            max_exit = center + cube_size_vector

            x_min, x_max = sorted(
                [(min_exit[0] - point_on_grid[0]) / ray[0], (max_exit[0] - point_on_grid[0]) / ray[0]])
            y_min, y_max = sorted(
                [(min_exit[1] - point_on_grid[1]) / ray[1], (max_exit[1] - point_on_grid[1]) / ray[1]])
            z_min, z_max = sorted(
                [(min_exit[2] - point_on_grid[2]) / ray[2], (max_exit[2] - point_on_grid[2]) / ray[2]])

            t_min = max(x_min, y_min, z_min)
            t_max = min(x_max, y_max, z_max)

            # Check if the ray intersects the cube
            if t_min < t_max:
                # Check if the intersection point is closer than the previous closest intersection point
                if 0.00001 < t_min < closest_intersection_distance:
                    # Calculate the point of intersection, set the closest intersection distance, and set the closest surface
                    closest_intersection_distance = t_min
                    closest_surface = (surface, point_on_grid + closest_intersection_distance * ray)

    # Check if the closest surface is the same as the original intersection point (with a small margin of error)
    is_hit = True
    for coord in range(3):
        if abs(closest_surface[1][coord] - original_intersection_point[coord]) > 0.00001:
            is_hit = False
            break

    return is_hit

if __name__ == '__main__':
    main()
