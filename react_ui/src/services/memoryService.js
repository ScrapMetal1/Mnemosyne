/* src/services/memoryService.js */
const API_BASE = 'http://127.0.0.1:5000/api'

export const memoryService = {  // this is memory related tasks package
    async captureMemory() {
        try {
            const response = await fetch(`${API_BASE}/memory/capture`, {
                method: "POST",
            })

            if (!response.ok) {
                throw new Error(`Memory capture failed: ${response.statusText}`)
            }

            const data = await response.json()
            return data
        } catch (error) {
            console.error("Error capturing memory:", error)
            throw error // so the component know its failed. Not really necessary here. 
        }
    }
}
