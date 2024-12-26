#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <output_file> <expected_file>\n", argv[0]);
        return 1;
    }

    FILE *output_file = fopen(argv[1], "r");
    FILE *expected_file = fopen(argv[2], "r");

    if (output_file == NULL || expected_file == NULL) {
        perror("Failed to open file");
        return 1;
    }

    int output_value, expected_value;
    while (fscanf(output_file, "%d", &output_value) == 1 && fscanf(expected_file, "%d", &expected_value) == 1) {
        if (output_value != expected_value) {
            fclose(output_file);
            fclose(expected_file);
            return 1; // Files are different
        }
    }

    // Check if both files reached EOF
    if (fscanf(output_file, "%d", &output_value) == 1 || fscanf(expected_file, "%d", &expected_value) == 1) {
        fclose(output_file);
        fclose(expected_file);
        return 1; // One file has more data than the other
    }

    fclose(output_file);
    fclose(expected_file);
    return 0; // Files are the same
}
