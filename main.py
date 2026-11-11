import cv2
import numpy as np

# خواندن تصویر
image = cv2.imread("flower.jpg")

# تبدیل به خاکستری
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# فیلتر برای کاهش نویز
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# آستانه‌گذاری
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# پیدا کردن کانتورها
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# شمارش اشیا
object_count = len(contours)
print("تعداد اشیا در تصویر:", object_count)

# کشیدن کانتورها روی تصویر
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# نوشتن تعداد روی تصویر
cv2.putText(output, f"Count: {object_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 0, 255), 2)

# نمایش نتایج
cv2.imshow("Original", image)
cv2.imshow("Threshold", thresh)
cv2.imshow("Objects Detected", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
