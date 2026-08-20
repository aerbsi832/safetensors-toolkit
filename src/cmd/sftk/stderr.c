#include <stdio.h>
#include <stdint.h>

void sftk_stderr_write(const uint8_t *bytes, int32_t length) {
  if (bytes != NULL && length > 0) {
    (void)fwrite(bytes, 1, (size_t)length, stderr);
  }
}
