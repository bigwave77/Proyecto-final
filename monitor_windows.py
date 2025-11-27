import cv2
import numpy as np
import time
import ctypes

class VenMonitor:
    """
    Monitor de ventana activa.
    Detecta si el usuario cambia de ventana.
    """
    def __init__(self, window_title_to_watch):
        self.target_title = window_title_to_watch
        self.user32 = ctypes.windll.user32
        self.is_window_active = True
        
    def check_status(self):
        """
        True si la ventana del examen es la activa.
        False si el usuario cambió de ventana.
        """
        # Obtener el manejador de la ventana que está en primer plano (activa)
        foreground_hwnd = self.user32.GetForegroundWindow()
        
        # Obtener la longitud del título
        length = self.user32.GetWindowTextLengthW(foreground_hwnd)
        
        # Crear buffer y obtener el texto
        buf = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(foreground_hwnd, buf, length + 1)
        
        active_title = buf.value
        
        # Comprobar si nuestra ventana de examen es la activa
        if self.target_title in active_title:
            self.is_window_active = True
        else:
            self.is_window_active = False
            
        return self.is_window_active


class Tracker:
    def __init__(self):
        # Cargar Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Parámetros Lucas-Kanade
        self.lk_params = dict(winSize=(21, 21),  
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Parámetros para encontrar puntos (Shi-Tomasi)
        self.feature_params = dict(maxCorners=50,        
                                   qualityLevel=0.1,     
                                   minDistance=15,       
                                   blockSize=7)
        
        self.old_gray = None
        self.p0 = None
        self.face_center_initial = None
        self.current_direction = "Centro"
        
        # Control de re-detección
        self.frame_count = 0
        self.redetect_interval = 15 

    def detect_face_initial(self, frame_gray):
        """Detecta la cara y selecciona puntos solo en el centro"""
        faces = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        
        if len(faces) > 0:
            # Tomar la cara más grande
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            (x, y, w, h) = faces[0]
            
            # Máscara para seleccionar puntos solo en la zona central del rostro
            mask = np.zeros_like(frame_gray)
            roi_x = x + int(w * 0.25)      
            roi_y = y + int(h * 0.30)      
            roi_w = int(w * 0.50)          
            roi_h = int(h * 0.50)          
            
            mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
            
            # Obtener puntos característicos
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            
            if self.p0 is not None:
                self.old_gray = frame_gray.copy()
                self.face_center_initial = (x + w//2, y + h//2)
                self.current_direction = "Centro" 
                return (x, y, w, h)
        
        return None

    def track(self, frame_gray):
        self.frame_count += 1
        
        # Re-detectar cada ciertos frames para corregir deriva
        if self.p0 is None or self.frame_count % self.redetect_interval == 0:
            rect = self.detect_face_initial(frame_gray)
            if rect: return rect
            if self.p0 is None: return None

        # Calcular Flujo Óptico
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            
            if len(good_new) < 5:
                self.p0 = None
                return None

            # Calcular centro actual basado en el movimiento promedio de los puntos
            curr_x = np.mean(good_new[:, 0])
            curr_y = np.mean(good_new[:, 1])
            
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            
            if self.face_center_initial is not None:
                threshold_x = 60  
                threshold_y = 40  
                
                dx = curr_x - self.face_center_initial[0]
                dy = curr_y - self.face_center_initial[1]

                direction = "Centro"
                
                # Determinar dirección
                if dx > threshold_x: direction = "Derecha" 
                elif dx < -threshold_x: direction = "Izquierda"
                elif dy > threshold_y: direction = "Abajo"
                elif dy < -threshold_y: direction = "Arriba"
                
                self.current_direction = direction

            box_size = 160
            return (int(curr_x) - box_size//2, 
                    int(curr_y) - box_size//2, 
                    box_size, box_size)
            
        return None


class ExamApp:
    def __init__(self):
        self.window_name = "Monitor de Examen"
        self.cap = cv2.VideoCapture(0)
        self.tracker = Tracker()
        self.win_monitor = VenMonitor(self.window_name)
        
        # Estadísticas
        self.exam_running = False
        self.start_time = 0
        self.total_time = 0
        self.attention_time = 0
        self.distraction_stats = {
            "Izquierda": 0.0, "Derecha": 0.0, 
            "Arriba": 0.0, "Abajo": 0.0, 
            "Cambio de Ventana": 0.0
        }
        self.last_frame_time = 0
        self.suspicious_threshold = 0.40 # 40%

    def draw_ui(self, frame):
        # Dibujar botón de estado
        color = (0, 255, 0) if self.exam_running else (0, 0, 255)
        text = "DETENER EXAMEN (Clic)" if self.exam_running else "INICIAR EXAMEN (Clic)"
        cv2.rectangle(frame, (10, 10), (280, 60), color, -1)
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar estado actual
        if self.exam_running:
            # Estado de Tracking
            status_color = (255, 0, 0)
            status_text = f"Cabeza: {self.tracker.current_direction}"
            
            # Alerta de Ventana
            if not self.win_monitor.is_window_active:
                status_text = "ALERTA: VENTANA INACTIVA"
                status_color = (0, 0, 255)
            
            cv2.putText(frame, status_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Timer
            elapsed = time.time() - self.start_time
            cv2.putText(frame, f"Tiempo: {elapsed:.1f}s", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Coordenadas del botón
            if 10 <= x <= 280 and 10 <= y <= 60:
                if not self.exam_running:
                    self.start_exam()
                else:
                    self.stop_exam()

    def start_exam(self):
        print("Examen Iniciado")
        self.exam_running = True
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        # Reiniciar estadísticas
        self.attention_time = 0
        self.distraction_stats = {k: 0.0 for k in self.distraction_stats}
        # Cerrar ventana de reporte si existiera de un examen anterior
        try:
            cv2.destroyWindow("Resultados del Examen")
        except:
            pass

    def stop_exam(self):
        self.exam_running = False
        self.total_time = time.time() - self.start_time
        self.generate_report_console()
        self.report()

    def get_stats_calc(self):
        total_distraction = sum(self.distraction_stats.values())
        real_attention = max(0, self.total_time - total_distraction)
        perc_distraction = (total_distraction / self.total_time) * 100 if self.total_time > 0 else 0
        is_suspicious = perc_distraction > (self.suspicious_threshold * 100)
        return total_distraction, real_attention, perc_distraction, is_suspicious

    def report(self):
        """Genera y muestra una imagen con las estadísticas del examen."""
        total_distraction, real_attention, perc_distraction, is_suspicious = self.get_stats_calc()
        
        # Crear imagen negra (lienzo)
        height, width = 500, 600
        report_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Colores y Fuentes
        white = (255, 255, 255)
        green = (0, 255, 0)
        red = (0, 0, 255)
        yellow = (0, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Dibujar textos
        y = 50
        cv2.putText(report_img, "REPORTE FINAL DEL EXAMEN", (100, y), font, 0.8, white, 2)
        
        y += 50
        cv2.putText(report_img, f"Duracion Total: {self.total_time:.2f} s", (30, y), font, 0.7, white, 1)
        y += 40
        cv2.putText(report_img, f"Tiempo Atencion: {real_attention:.2f} s", (30, y), font, 0.7, green, 1)
        y += 40
        cv2.putText(report_img, f"Tiempo Distraccion: {total_distraction:.2f} s", (30, y), font, 0.7, red, 1)
        y += 40
        cv2.putText(report_img, f"Porcentaje Distraccion: {perc_distraction:.1f}%", (30, y), font, 0.7, red, 1)
        
        y += 50
        cv2.putText(report_img, "Detalle de Distracciones:", (30, y), font, 0.6, yellow, 1)
        y += 30
        for k, v in self.distraction_stats.items():
            if v > 0:
                cv2.putText(report_img, f"- {k}: {v:.2f} s", (50, y), font, 0.6, white, 1)
                y += 25
        
        y += 40
        if is_suspicious:
            cv2.putText(report_img, "RESULTADO: SOSPECHOSO", (30, y), font, 1.0, red, 2)
        else:
            cv2.putText(report_img, "RESULTADO: NORMAL", (30, y), font, 1.0, green, 2)

        cv2.imshow("Resultados del Examen", report_img)

    def generate_report_console(self):
        total_distraction, real_attention, perc_distraction, is_suspicious = self.get_stats_calc()
        
        print("REPORTE FINAL")
        print("\n")
        print(f"Duración Total: {self.total_time:.2f} seg")
        print(f"Tiempo Atención: {real_attention:.2f} seg")
        print(f"Tiempo Distracción: {total_distraction:.2f} seg ({perc_distraction:.2f}%)")
        print("\n")
        print("Detalle de Distracciones:")
        for k, v in self.distraction_stats.items():
            print(f"  - {k}: {v:.2f} seg")
        print("\n")
        if is_suspicious:
            print("RESULTADO: COMPORTAMIENTO SOSPECHOSO")
        else:
            print("RESULTADO: Comportamiento Normal")


    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.handle_click)

        print("Aplicación lista. Ajusta tu cámara y presiona INICIAR EXAMEN.")

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # Voltear frame (espejo)
            frame = cv2.flip(frame, 1)
            # Resize opcional si la cámara es muy grande
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Lógica de Tracking
            face_rect = self.tracker.track(gray)
            
            if self.exam_running:
                current_time = time.time()
                dt = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                # 1. Revisar Ventana Activa
                is_active = self.win_monitor.check_status()
                
                if not is_active:
                    self.distraction_stats["Cambio de Ventana"] += dt
                # 2. Revisar Dirección de Cabeza
                elif self.tracker.current_direction != "Centro":
                    self.distraction_stats[self.tracker.current_direction] += dt
                else:
                    self.attention_time += dt

            # Visualización
            if face_rect is not None:
                (x, y, w, h) = face_rect
                # Color del recuadro: Verde (OK), Rojo (Distraído), Amarillo (Ventana Inactiva)
                if self.exam_running and not self.win_monitor.is_window_active:
                    color = (0, 255, 255) # Amarillo
                elif self.tracker.current_direction == "Centro":
                    color = (0, 255, 0)   # Verde
                else:
                    color = (0, 0, 255)   # Rojo
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Dibujar puntos de Optical Flow
                if self.tracker.p0 is not None:
                    for i, point in enumerate(self.tracker.p0):
                        a, b = point.ravel()
                        cv2.circle(frame, (int(a), int(b)), 2, (0, 255, 255), -1)

            self.draw_ui(frame)
            
            cv2.imshow(self.window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC para salir
                if self.exam_running:
                    self.stop_exam()
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ExamApp()
    app.run()